// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!


#include <glibmm.h>

#include <gtkmm/gesturepan.h>
#include <gtkmm/private/gesturepan_p.h>


/* Copyright (C) 2014 The gtkmm Development Team
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 */

#include <gtk/gtk.h>

namespace
{


static void GesturePan_signal_pan_callback(GtkGesturePan* self, GtkPanDirection p0,gdouble p1,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,PanDirection,double > SlotType;

  auto obj = dynamic_cast<GesturePan*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(((PanDirection)(p0))
, p1
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo GesturePan_signal_pan_info =
{
  "pan",
  (GCallback) &GesturePan_signal_pan_callback,
  (GCallback) &GesturePan_signal_pan_callback
};


} // anonymous namespace

// static
GType Glib::Value<Gtk::PanDirection>::value_type()
{
  return gtk_pan_direction_get_type();
}


namespace Glib
{

Glib::RefPtr<Gtk::GesturePan> wrap(GtkGesturePan* object, bool take_copy)
{
  return Glib::RefPtr<Gtk::GesturePan>( dynamic_cast<Gtk::GesturePan*> (Glib::wrap_auto ((GObject*)(object), take_copy)) );
  //We use dynamic_cast<> in case of multiple inheritance.
}

} /* namespace Glib */


namespace Gtk
{


/* The *_Class implementation: */

const Glib::Class& GesturePan_Class::init()
{
  if(!gtype_) // create the GType if necessary
  {
    // Glib::Class has to know the class init function to clone custom types.
    class_init_func_ = &GesturePan_Class::class_init_function;

    // This is actually just optimized away, apparently with no harm.
    // Make sure that the parent type has been created.
    //CppClassParent::CppObjectType::get_type();

    // Create the wrapper type, with the same class/instance size as the base type.
    register_derived_type(gtk_gesture_pan_get_type());

    // Add derived versions of interfaces, if the C type implements any interfaces:

  }

  return *this;
}


void GesturePan_Class::class_init_function(void* g_class, void* class_data)
{
  const auto klass = static_cast<BaseClassType*>(g_class);
  CppClassParent::class_init_function(klass, class_data);


}


Glib::ObjectBase* GesturePan_Class::wrap_new(GObject* object)
{
  return new GesturePan((GtkGesturePan*)object);
}


/* The implementation: */

GtkGesturePan* GesturePan::gobj_copy()
{
  reference();
  return gobj();
}

GesturePan::GesturePan(const Glib::ConstructParams& construct_params)
:
  GestureDrag(construct_params)
{

}

GesturePan::GesturePan(GtkGesturePan* castitem)
:
  GestureDrag((GtkGestureDrag*)(castitem))
{}


GesturePan::GesturePan(GesturePan&& src) noexcept
: GestureDrag(std::move(src))
{}

GesturePan& GesturePan::operator=(GesturePan&& src) noexcept
{
  GestureDrag::operator=(std::move(src));
  return *this;
}

GesturePan::~GesturePan() noexcept
{}


GesturePan::CppClassType GesturePan::gesturepan_class_; // initialize static member

GType GesturePan::get_type()
{
  return gesturepan_class_.init().get_type();
}


GType GesturePan::get_base_type()
{
  return gtk_gesture_pan_get_type();
}


GesturePan::GesturePan()
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  GestureDrag(Glib::ConstructParams(gesturepan_class_.init()))
{
  

}

GesturePan::GesturePan(Widget& widget, Orientation orientation)
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  GestureDrag(Glib::ConstructParams(gesturepan_class_.init(), "widget", (widget).gobj(), "orientation", ((GtkOrientation)(orientation)), nullptr))
{
  

}

Glib::RefPtr<GesturePan> GesturePan::create(Widget& widget, Orientation orientation)
{
  return Glib::RefPtr<GesturePan>( new GesturePan(widget, orientation) );
}

Orientation GesturePan::get_orientation() const
{
  return ((Orientation)(gtk_gesture_pan_get_orientation(const_cast<GtkGesturePan*>(gobj()))));
}

void GesturePan::set_orientation(Orientation orientation)
{
  gtk_gesture_pan_set_orientation(gobj(), ((GtkOrientation)(orientation)));
}


Glib::SignalProxy2< void,PanDirection,double > GesturePan::signal_pan()
{
  return Glib::SignalProxy2< void,PanDirection,double >(this, &GesturePan_signal_pan_info);
}


Glib::PropertyProxy< Orientation > GesturePan::property_orientation() 
{
  return Glib::PropertyProxy< Orientation >(this, "orientation");
}

Glib::PropertyProxy_ReadOnly< Orientation > GesturePan::property_orientation() const
{
  return Glib::PropertyProxy_ReadOnly< Orientation >(this, "orientation");
}


} // namespace Gtk

