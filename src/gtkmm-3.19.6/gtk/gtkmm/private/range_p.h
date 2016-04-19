// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_RANGE_P_H
#define _GTKMM_RANGE_P_H


#include <gtkmm/private/widget_p.h>

#include <glibmm/class.h>

namespace Gtk
{

class Range_Class : public Glib::Class
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef Range CppObjectType;
  typedef GtkRange BaseObjectType;
  typedef GtkRangeClass BaseClassType;
  typedef Gtk::Widget_Class CppClassParent;
  typedef GtkWidgetClass BaseClassParent;

  friend class Range;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  const Glib::Class& init();


  static void class_init_function(void* g_class, void* class_data);

  static Glib::ObjectBase* wrap_new(GObject*);

protected:

  //Callbacks (default signal handlers):
  //These will call the *_impl member methods, which will then call the existing default signal callbacks, if any.
  //You could prevent the original default signal handlers being called by overriding the *_impl method.
  static void value_changed_callback(GtkRange* self);
  static void adjust_bounds_callback(GtkRange* self, gdouble p0);
  static void move_slider_callback(GtkRange* self, GtkScrollType p0);
  static gboolean change_value_callback(GtkRange* self, GtkScrollType p0, gdouble p1);

  //Callbacks (virtual functions):
  static void get_range_border_vfunc_callback(GtkRange* self, GtkBorder* border);
};


} // namespace Gtk


#endif /* _GTKMM_RANGE_P_H */
