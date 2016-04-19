// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_CELLRENDERERACCEL_P_H
#define _GTKMM_CELLRENDERERACCEL_P_H


#include <gtkmm/private/cellrenderertext_p.h>

#include <glibmm/class.h>

namespace Gtk
{

class CellRendererAccel_Class : public Glib::Class
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef CellRendererAccel CppObjectType;
  typedef GtkCellRendererAccel BaseObjectType;
  typedef GtkCellRendererAccelClass BaseClassType;
  typedef Gtk::CellRendererText_Class CppClassParent;
  typedef GtkCellRendererTextClass BaseClassParent;

  friend class CellRendererAccel;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  const Glib::Class& init();


  static void class_init_function(void* g_class, void* class_data);

  static Glib::ObjectBase* wrap_new(GObject*);

protected:

  //Callbacks (default signal handlers):
  //These will call the *_impl member methods, which will then call the existing default signal callbacks, if any.
  //You could prevent the original default signal handlers being called by overriding the *_impl method.
  static void accel_edited_callback(GtkCellRendererAccel* self, const gchar* p0, guint p1, GdkModifierType p2, guint p3);
  static void accel_cleared_callback(GtkCellRendererAccel* self, const gchar* p0);

  //Callbacks (virtual functions):
};


} // namespace Gtk


#endif /* _GTKMM_CELLRENDERERACCEL_P_H */

